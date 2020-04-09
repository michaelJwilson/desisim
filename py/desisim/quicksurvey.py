'''
desisim.quicksurvey
===================

Code for quickly simulating the survey results given a mock catalog and
a list of tile epochs to observe.

Directly depends on the following DESI products:

* desitarget.mtl
* :mod:`desisim.quickcat`
* `fiberassign <https://github.com/desihub/fiberassign>`_
'''

from   __future__ import absolute_import, division, print_function
import gc
import sys
import numpy as np
import os
import json
from   datetime import datetime
import shutil
import glob
import subprocess
from   astropy.table import Table, Column
import os.path
from   collections import Counter
from   time import time, asctime
import fitsio
import desimodel
import desitarget.mtl
import warnings
from   desisim.quickcat import quickcat
from   astropy.table import join
from   desitarget.targetmask import desi_mask
from   desimodel.footprint import is_point_in_desi
from   fitsio import FITS
from   pathlib import Path

warnings.simplefilter(action='error', category=FutureWarning)

def inprogram(program, programs):
    program = program.upper()
    
    if program == 'BRIGHT':
        return  programs == 'BRIGHT'

    else:
        return  (programs == 'DARK') | (programs == 'GRAY')

class SimSetup(object):
    """
    Setup to simulate the DESI survey

    Attributes:
        output_path (str): Path to write the outputs.x
        targets_path (str): Path where the files targets.fits can be found
        epochs_path (str): Path where the epoch files can be found.
        fiberassign (str): Name of the fiberassign script  
        template_fiberassign (str): Filename of the template input for fiberassign
        n_epochs (int): number of epochs to be simulated.

    """
    def __init__(self, output_path, targets_path, fiberassign, exposures, fiberassignlog, footprint, program='dark'):
        """
        Initializes all the paths, filenames and numbers describing DESI survey.

        Args:
            output_path (str): Path to write the outputs.x
            targets_path (str): Path where the files targets.fits can be found
            fiberassign (str): Name of the fiberassign executable
            template_fiberassign (str): Filename of the template input for fiberassign
            exposures (stri): exposures.fits file summarazing surveysim results
            fiberassignlog (str): .fits file with the fiberassign run log.
        """

        print('\n\n--- DESI E2E {} ---'.format(program.upper()))
        
        self.output_path = output_path + '/{}/'.format(program)
        self.targets_path = targets_path
        self.fiberassign = fiberassign  
        self.exposures = Table(fitsio.read(exposures, upper=True))
        self.tmp_output_path = os.path.join(self.output_path, 'tmp/')
        self.tmp_fiber_path = os.path.join(self.tmp_output_path, 'fiberassign/')
        self.surveyfile = os.path.join(self.tmp_output_path, 'survey_list.txt')
        self.skyfile  = os.path.join(self.targets_path,'skies.fits')
        self.stdfile  = os.path.join(self.targets_path,'std.fits')
        self.truthfile  = os.path.join(self.targets_path,'truth-{}.fits'.format(program))
        self.targetsfile = os.path.join(self.targets_path,'targets-{}.fits'.format(program))
        self.fibstatusfile = os.path.join(self.targets_path,'fiberstatus.ecsv')
        self.footprintfile = footprint
        self.zcat_file = None
        self.mtl_file = None
        self.program=program
        
        # List of tiles fiberassigned on this epoch. 
        self.epoch_tiles = list()
        # List of tiles completed during this epoch. 
        self.epoch_ctiles = list()
        self.atilefiles = list()
        self.ctilefiles = list()
        self.plan_tiles = list()
        self.observed_tiles = list()
        self.epochs_list = list()
        self.n_epochs = 0
        self.start_epoch = 0
        self.summary = {}
        
        os.environ['DESI_LOGLEVEL'] = 'DEBUG'
        
        ##
        tiles      = Table(desimodel.io.load_tiles(tilesfile=footprint, cache=False))
        
        assignlog  = Table(fitsio.read(fiberassignlog))

        ##  Restrict assigned tiles to a given program.
        def _(x):
            x          = join(x, tiles['TILEID', 'PROGRAM'], join_type='left', keys='TILEID')
            
            isin       = inprogram(self.program, x['PROGRAM'])
            x          = x[isin]

            return x
        
        assignlog      = _(assignlog)

        self.exposures = _(self.exposures)
        self.exposures.sort('MJD')
                
        ##  .decode('utf-8')
        ids        = assignlog['TILEID']
        dates      = list(assignlog['ASSIGNDATE'])
        dates      = [date.replace('-', '') for date in dates]
        dates      = np.array(dates)
        
        udates     = [str(x) for x in np.unique(dates)]
        udates.remove('99999999')
        
        self.assigndates = np.array([x[0:4] + '-' + x[4:6] + '-' + x[6:8] + 'T12:00:00' for x in udates])
        
        ##  Udates must be chronologically increasing.  
        ##  assert  np.all(np.diff([np.int(x) for x in dates]) > 0) 
        
        self.n_epochs = len(udates)

        ##  Must only be completed exposures.                                                                                                                                                                                     
        assert  np.all(self.exposures['SNR2FRAC'] > 1.0)

        ##  And ordered in time.
        assert	np.all(np.diff(self.exposures['MJD']) > 0.0)

        ##  Either BRIGHT only, or DARK AND GRAY.
        ##  assert  TODO
        
        self.lastnight = self.exposures['NIGHT'][-1]

        tmp       =  udates + [self.lastnight]
        
        edts      =  [datetime.strptime(e, "%Y%m%d") for e in self.exposures['NIGHT']]

        #
        self.exposures['EPOCH'] = -1 * np.ones_like(self.exposures['TILEID'])
        
        for i, udate in enumerate(udates):
          ii      =  (dates == udate)

          ##  Contains all assigned tiles, not just those completed.  
          self.epoch_tiles.append(ids[ii].tolist())

          adt     =  datetime.strptime(udate,    "%Y%m%d")
          adtup   =  datetime.strptime(tmp[i+1], "%Y%m%d")

          isin    =  [(edt >= adt) & (edt < adtup) for edt in edts]

          ##  Contains all tiles completed in this epoch.                                                                                                                                                              
          self.epoch_ctiles.append(np.array(self.exposures['TILEID'][isin]).tolist())

          self.exposures['EPOCH'][isin] = i
          
          # print(udate, tmp[i+1], np.array(self.exposures['TILEID'][isin]).tolist())

          self.summary[udate]               = {}
          
          self.summary[udate]['NASSIGNED']  = len(self.epoch_tiles[i])
          self.summary[udate]['NCOMPLETED'] = len(self.epoch_ctiles[i])
          
          print('\nTiles assigned on date {} (start of epoch {: 3d}): {}'.format(udate, i, self.summary[udate]['NASSIGNED']))
          print('Tiles completed during this epoch ({: 2d}): {}'.format(i, self.summary[udate]['NCOMPLETED']))
          
        ##  Deal with the last night.
        self.exposures['EPOCH'][self.exposures['NIGHT'] == self.lastnight]  = i
        self.epoch_ctiles[-1] += np.array(self.exposures['TILEID'][self.exposures['NIGHT'] == self.lastnight]).tolist()
        self.summary[udates[-1]]['NCOMPLETED'] = len(self.epoch_ctiles[-1])

        self.summary['UNASSIGNED']  = np.count_nonzero(dates == '99999999')
        self.summary['UNCOMPLETED'] = len(np.concatenate(self.epoch_tiles)) - len(np.concatenate(self.epoch_ctiles))
        
        print('\nRemaining unassigned tiles: {}.'.format(self.summary['UNASSIGNED']))
        print('\nRemaining uncompleted tiles: {}.\n'.format(self.summary['UNCOMPLETED']))

        ##  Write exposures reduced, with epoch of completion.
        self.exposures.write(self.output_path + '/epochs-{}.fits'.format(self.program), format='fits', overwrite=True)

        ## 
        ids        = np.concatenate(self.epoch_tiles)
        
        # self.exposures.pprint(max_lines=-1)
        
        ##  Include all tiles that were assigned, not just completed. 
        isin       = np.isin(tiles['TILEID'], ids)

        self.tiles = tiles[isin]
        
    def create_directories(self):
        """
        Creates output directories to store simulation results.
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if not os.path.exists(self.tmp_output_path):
            os.makedirs(self.tmp_output_path)

        if not os.path.exists(self.tmp_fiber_path):
            os.makedirs(self.tmp_fiber_path)

    def cleanup_directories(self):
        """
        Deletes files in the temporary output directory
        """
        if os.path.exists(self.tmp_output_path):
            shutil.rmtree(self.tmp_output_path)

    def epoch_data_exists(self, epoch_id=0):
        """
        Check epoch directory for zcat.fits and mtl.fits files.
        """
        backup_path = os.path.join(self.output_path, '{}'.format(epoch_id))

        mtl_file    = os.path.join(backup_path, 'mtl-{}.fits'.format(self.program))
        zcat_file   = os.path.join(backup_path, 'zcat-{}.fits'.format(self.program))
        
        if os.path.isfile(mtl_file) and os.path.isfile(zcat_file):
            return  True

        else:
            return  False
      
    def backup_epoch_data(self, epoch_id=0):
        """
        Deletes files in the temporary output directory and backup b/d mtls, zcat
        and fiberassign dir.  

        Args:
            epoch_id (int): Epoch's ID to backup/copy from the output directory.

        """
        backup_path = os.path.join(self.output_path, '{}'.format(epoch_id))

        # keep a copy of zcat.fits
        if not os.path.exists(backup_path):
          os.makedirs(backup_path)

        # keep a copy of mtl.fits
        shutil.copy(self.mtl_file,  backup_path)
        shutil.copy(self.zcat_file, backup_path)
        
        # keep a copy of all the fiberassign files
        fiber_backup_path = os.path.join(backup_path, 'fiberassign')

        if not os.path.exists(fiber_backup_path):
          os.makedirs(fiber_backup_path)

        for tt in self.atilefiles:
          shutil.copy(tt, fiber_backup_path)

    def create_surveyfile(self, epoch):
        """
        Creates text file of tiles survey_list.txt to be used by fiberassign

        Args:
            epoch (int) : epoch of tiles to write

        Notes:
            The file is written to the temporary directory in self.tmp_output_path
        """

        # create survey list from mtl_epochs IDS
        surveyfile  = os.path.join(self.tmp_output_path, "survey_list.txt")

        # Already restricted to a given program.
        tiles       = self.tiles
        etiles      = self.epoch_tiles[epoch]

        isin        = np.isin(tiles['TILEID'], etiles)

        # Tiles for this assignment epoch. 
        tiles       = tiles[isin]

        atiles      = np.array(tiles['TILEID'])
            
        np.savetxt(surveyfile, atiles, fmt='%d')

        self.atilefiles = []
        
        for assigned in atiles:
          tilename = self.tmp_fiber_path + 'fiberassign-{:06d}.fits'.format(assigned)

          self.atilefiles.append(tilename)
          
        print("{} {} tiles to be included in fiberassign".format(len(tiles), self.program))

        self.summary['FIBERASSIGN_{}_TILES_{}'.format(self.program, epoch)] = np.array(tiles['TILEID']).tolist()
        
        return  len(tiles)
        
    def update_observed_tiles(self, epoch):
        """
        Creates the list of fiberassign-{} files to be gathered to build the redshift catalog.
        """        
        # 
        tiles  = self.tiles
        
        # Only gather tiles in zcat which are both assigned and complete.
        ctiles = np.array(self.epoch_ctiles[epoch])
        
        # those assigned this epoch.
        syncd  = ctiles[ np.isin(ctiles, self.epoch_tiles[epoch])]
        nsyncd = ctiles[~np.isin(ctiles, self.epoch_tiles[epoch])]

        self.summary['COMPLETE_{}'.format(epoch)] = ctiles.tolist()

        self.ctilefiles = []
        
        # Tiles that are synced to this epoch. 
        for i in syncd:
            tilename    = os.path.join(self.tmp_fiber_path, 'fiberassign-{:06d}.fits'.format(i))

            # retain ability to use previous version of tile files
            oldtilename = os.path.join(self.tmp_fiber_path, 'tile-%05d.fits'.format(i))

            if os.path.isfile(tilename):
                self.ctilefiles.append(tilename)

            elif os.path.isfile(oldtilename):
                self.ctilefiles.append(oldtilename)

            else:
              print('Suggested but does not exist {}'.format(tilename))

        # Tiles completed this epoch, but assigned in a previous epoch.
        for i in nsyncd:
          # Epoch this tile was assigned. 
          for epoch, assigned in enumerate(self.epoch_tiles):
             if i in assigned:
                 break
             
          tilename = self.output_path + '/{}/fiberassign/fiberassign-{:06d}.fits'.format(epoch, i)

          # retain ability to use previous version of tile files                                                                                                                                                                 
          oldtilename = self.output_path + '/{}/fiberassign/tile-{:05d}.fits'.format(epoch, i)
          
          if os.path.isfile(tilename):
            self.ctilefiles.append(tilename)

          else:
            print('Suggested but does not exist {}'.format(tilename))

            exit(0)
            
        print("{} {} synced tiles and {} non-synced tiles to gather for zcat.  Found {}.".format(asctime(), len(syncd), len(nsyncd), len(self.ctilefiles)))
        
    def simulate_epoch(self, epoch, truth, targets, exposures, perfect=False, zcat=[None, None]):
        """
        Core routine simulating a DESI epoch,

        Args:
            epoch (int): epoch to simulate
            perfect (boolean): Default: False. Selects whether how redshifts are taken from the truth file.
                True: redshifts are taken without any error from the truth file.
                False: redshifts include uncertainties.
            truth (Table): Truth data
            targets (Table): Targets data
            zcat (Table): Redshift Catalog Data
        
        Notes:
            This routine simulates three steps:
            * Merged target list creation
            * Fiber allocation
            * Redshift catalogue construction
        """
 
        # create the MTL file.
        print("{} Starting MTL".format(asctime()))

        self.mtl_file = os.path.join(self.tmp_output_path, 'mtl-{}.fits'.format(self.program))
          
        if epoch == 0:
          if self.program == 'dark': 
            mtl = desitarget.mtl.make_mtl(targets, obscon='DARK|GRAY')

          if self.program == 'bright':
            mtl = desitarget.mtl.make_mtl(targets, obscon='BRIGHT')
          
        else:
          if self.program == 'dark':
            mtl = desitarget.mtl.make_mtl(targets, obscon='DARK|GRAY', zcat=zcat)

          if self.program == 'bright':
            mtl = desitarget.mtl.make_mtl(targets, obscon='BRIGHT', zcat=zcat)

        print("{} Writing MTL".format(asctime()))

        # fits = FITS(self.mtl_file, 'rw') 
        # fits.write(mtl.as_array())
        # fits.close()
        
        mtl.write(self.mtl_file, overwrite=True)
          
        del mtl
         
        gc.collect()
        
        print("{} Finished MTLs.".format(asctime()))
                
        # clean files and prepare fiberasign inputs
        tilefiles = sorted(glob.glob(self.tmp_fiber_path+'/tile*.fits'))

        if tilefiles:
            for tilefile in tilefiles:
                os.remove(tilefile)

        # setup the tileids for the current observation epoch
        ntodo = self.create_surveyfile(epoch)
        
        if np.any(ntodo > 0):        
          # launch fiberassign
          print("{} Launching fiberassign".format(asctime()))

          ##  f = open('fiberassign.log', 'a')

          # '--stdstar',  self.stdfile  '--fibstatusfile',  self.fibstatusfile],
          # '--overwrite'

          # log.warning('Not overwriting fiberassign output')

          def call_fiberassign(mtl_path, surveyfile, overwrite=False):
            cmd = [self.fiberassign,
                   '--mtl',  mtl_path,
                   '--sky',  self.skyfile,
                   '--surveytiles',  surveyfile,
                   '--rundate', self.assigndates[epoch],
                   '--outdir', os.path.join(self.tmp_output_path, 'fiberassign')]

            if overwrite:
                cmd = cmd + ['--overwrite']

            if self.footprintfile is not None:
                cmd = cmd + ['--footprint', self.footprintfile]
          
            cmd = ' '.join(x for x in cmd)
           
            print(cmd)

            # Print real-time fiberassign output to screen. 
            p     = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE) 

            while True:
              out = p.stderr.read(1)
              out = out.decode('utf-8')
              
              if out == '' and p.poll() != None:
                  break

              if out != '':
                  sys.stdout.write(out)
                  sys.stdout.flush()
          
            if overwrite & (p.returncode > 0):
              raise ValueError('Assignment error.')

            return  cmd
        
          mtl_path = os.path.join(self.tmp_output_path, 'mtl-{}.fits'.format(self.program))
                        
          cmd      = call_fiberassign(mtl_path, self.surveyfile)

          self.summary['FIBERASSIGN_{}_CMD_{}'.format(self.program, epoch)] = cmd
              
          print("{} Finished fiberassign".format(asctime()))

          ##  f.close()

        else:
          print("Skipping fiberassign for {} tiles.".format(ntodo))
          
        # create a list of fiberassign tiles to read and update zcat: must have been completed
        # according to exposures. 
        self.update_observed_tiles(epoch)

        print("{} starting quickcat".format(asctime()))

        def write_zcat(targets, truth, zcat, filename):
          # write the zcat, it uses the tilesfiles constructed in the last step.
          zcat_file          = os.path.join(self.tmp_output_path, filename)

          newzcat, fibermaps = quickcat(self.ctilefiles, targets, truth, zcat=zcat,
                                        exposures=exposures, perfect=perfect,\
                                        summary=None)
          if fibermaps is not None:
            for key in fibermaps.keys():            
              (fpath, fmap)  = fibermaps[key]

              print('Writing {}.'.format(fpath))

              ## filepath is fibermap-EXPID.fits
              fpath          = self.output_path + '/{}/fiberassign/{}'.format(epoch, fpath)

              dirname        = os.path.dirname(fpath) 
 
              if not os.path.exists(fpath):
                Path(dirname).mkdir(parents=True, exist_ok=True)
         
                fmap.write(fpath, format='fits', overwrite=True)

            print("{} writing zcat".format(asctime()))

            newzcat.write(zcat_file, format='fits', overwrite=True)

          print("{} Finished zcat".format(asctime()))

          del newzcat

        # Write zcat. 
        filename = 'zcat-{}.fits'.format(self.program)
              
        write_zcat(targets, truth, zcat, filename)

        self.zcat_file = os.path.join(self.tmp_output_path, 'zcat-{}.fits'.format(self.program))
            
        gc.collect()

        return 


    def simulate(self):
        """
        Simulate the DESI setup described by a SimSetup object.
        """
        self.create_directories()

        exposures = self.exposures
        
        print('Reading truth.')
        
        truth    = fitsio.read(self.truthfile)
        
        print('Reading targets.')

        targets  = fitsio.read(self.targetsfile)
        
        # Cut dark to assigned tiles.        
        isin     = is_point_in_desi(self.tiles, targets['RA'], targets['DEC']) 

        targets  = targets[isin]
        truth    =   truth[isin] 
        
        #- Drop columns that aren't needed to save memory while manipulating
        # truth.remove_columns(['SEED', 'MAG', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'])
        # targets.remove_columns([ 'SHAPEEXP_R', 'SHAPEEXP_E1', 'SHAPEEXP_E2', 'SHAPEDEV_R',
        #                         'SHAPEDEV_E1', 'SHAPEDEV_E2', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
        #                         'MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2'])
        
        gc.collect()

        names = list(truth.dtype.names)
        
        if 'MOCKID' in names:
            names.remove('MOCKID')
            
            truth = truth[names]
            
        for epoch in range(self.start_epoch, self.n_epochs):
            print('\n\n--- Epoch {} ---'.format(epoch))
            
            if not self.epoch_data_exists(epoch_id=epoch):       
                # Initializes mtl and zcat
                if epoch == 0:
                    zcat = None
                    
                else:
                    print('INFO: Running Epoch {}'.format(epoch))
                    print('INFO: reading zcat from previous epoch')

                    epochdir = os.path.join(self.output_path, str(epoch - 1))

                    zcat     = Table.read(os.path.join(epochdir, 'zcat-{}.fits'.format(self.program)))
                    
                # Update mtl and zcat.
                self.simulate_epoch(epoch, truth, targets, exposures, perfect=True, zcat=zcat)

                # copy mtl and zcat to epoch directory
                self.backup_epoch_data(epoch_id=epoch)

                del zcat
                
                gc.collect()
                
            else:
                print('    COMPLETED    '.format(epoch))

        # Write summary log.
        with open(self.output_path + '/summary-{}.json'.format(self.program), 'w') as ff:
          json.dump(self.summary, ff)
                
        self.cleanup_directories()
        
def print_efficiency_stats(truth, mtl_initial, zcat):
    print('Overall efficiency')
    tmp_init = join(mtl_initial, truth, keys='TARGETID')
    total = join(zcat, tmp_init, keys='TARGETID')

    true_types = ['LRG', 'ELG', 'QSO']
    zcat_types = ['GALAXY', 'GALAXY', 'QSO']

    for true_type, zcat_type in zip(true_types, zcat_types):
        i_initial = ((tmp_init['DESI_TARGET'] & desi_mask.mask(true_type)) != 0) & (tmp_init['TRUESPECTYPE'] == zcat_type)
        i_final = ((total['DESI_TARGET'] & desi_mask.mask(true_type)) != 0) & (total['SPECTYPE'] == zcat_type)
        n_t = 1.0*len(total['TARGETID'][i_final])
        n_i = 1.0*len(tmp_init['TARGETID'][i_initial])
        print("\t {} fraction : {}".format(true_type, n_t/n_i))
    #print("\t TRUE:ZCAT\n\t {}\n".format(Counter(zip(total['DESI_TARGET'], total['TYPE']))))
    return

def print_numobs_stats(truth, targets, zcat):
    print('Target distributions')
    #- truth and targets are row-matched, so directly add columns instead of join
    for colname in targets.colnames:
        if colname not in truth.colnames:
            truth[colname] = targets[colname]

    xcat = join(zcat, truth, keys='TARGETID')

    for times_observed in range(1,5):
        print('\t Fraction (number) with exactly {} observations'.format(times_observed))
        ii = (xcat['NUMOBS']==times_observed)
        c = Counter(xcat['DESI_TARGET'][ii])

        total = np.sum(list(c.values()))
        for k in c:
            print("\t\t {}: {} ({} total)".format(desi_mask.names(k), c[k]/total, c[k]))
    return

def efficiency_numobs_stats(_setup, epoch_id=0):
    backup_path = os.path.join(_setup.output_path, '{}'.format(epoch_id))
    backup_path_0 = os.path.join(_setup.output_path, '{}'.format(0))

    truth = Table.read(os.path.join(_setup.targets_path,'truth.fits'))
    targets = Table.read(os.path.join(_setup.targets_path,'targets.fits'))
    mtl0 = Table.read(os.path.join(backup_path_0,'mtl.fits'))
    zcat = Table.read(os.path.join(backup_path, 'zcat.fits'))

    print_efficiency_stats(truth, mtl0, zcat)
    print_numobs_stats(truth, targets, zcat)

    return

def summary_setup(_setup):
    for epoch in _setup.epochs_list:
        print('Summary for Epoch {}'.format(epoch))
        efficiency_numobs_stats(_setup, epoch_id = epoch)
