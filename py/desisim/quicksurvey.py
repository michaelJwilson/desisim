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
from   desisim.quickcat import quickcat
from   astropy.table import join
from   desitarget.targetmask import desi_mask
from   desimodel.footprint import is_point_in_desi
from   fitsio import FITS
from   pathlib import Path


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
    def __init__(self, output_path, targets_path, fiberassign, exposures, fiberassignlog, footprint):
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
        self.output_path = output_path
        self.targets_path = targets_path
        self.fiberassign = fiberassign  
        self.exposures = fitsio.read(exposures, upper=True)

        self.tmp_output_path = os.path.join(self.output_path, 'tmp/')
        self.tmp_fiber_path = os.path.join(self.tmp_output_path, 'fiberassign/')
        self.surveyfile = os.path.join(self.tmp_output_path, 'survey_list.txt')
        self.skyfile  = os.path.join(self.targets_path,'skies.fits')
        self.stdfile  = os.path.join(self.targets_path,'std.fits')
        self.truthfile  = os.path.join(self.targets_path,'truth.fits')
        self.targetsfile = os.path.join(self.targets_path,'targets.fits')
        self.fibstatusfile = os.path.join(self.targets_path,'fiberstatus.ecsv')
        self.footprintfile = footprint
        self.zcat_file = None
        self.mtl_file = None        
        self.epoch_tiles = list()
        self.tilefiles = list()
        self.plan_tiles = list()
        self.observed_tiles = list()
        self.epochs_list = list()
        self.n_epochs = 0
        self.start_epoch = 0
        
        os.environ['DESI_LOGLEVEL'] = 'DEBUG'
        
        ##
        tiles = desimodel.io.load_tiles()
        ids   = self.exposures['TILEID']

        isin  = np.isin(tiles['TILEID'], ids)
        
        self.tiles = tiles[isin]
        
        assignlog = fitsio.read(fiberassignlog)

        ## 
        ids       = assignlog['TILEID']
        dates     = list(assignlog['ASSIGNDATE'])
        dates     = [date.decode('utf-8').replace('-', '') for date in dates]
        dates     = np.array(dates)
        
        udates    = [str(x) for x in np.unique(dates)]
        udates.remove('99999999')

        self.assigndates = np.array([x[0:4] + '-' + x[4:6] + '-' + x[6:8] + 'T12:00:00' for x in udates])
        
        ##  Udates must be chronologically increasing.  
        ##  assert  np.all(np.diff([np.int(x) for x in dates]) > 0) 
        
        self.n_epochs = len(udates)
        
        for i, udate in enumerate(udates):
          ##  Must be completed. 
          ii      =  (dates == udate) & np.isin(ids, self.exposures['TILEID'])
          
          self.epoch_tiles.append(list(ids[ii]))
          
          print('Tiles assigned on date {} (start of epoch {: 3d}): {}'.format(udate, i, len(self.epoch_tiles[i])))

        print('\nRemaining unassigned tiles: {}.\n'.format(np.count_nonzero(dates == '99999999')))
          
        ##  Must only be completed exposures. 
        assert  np.all(self.exposures['SNR2FRAC'] > 1.0)                                       
                                         
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
        mtl_file = os.path.join(backup_path, 'mtl.fits')
        zcat_file = os.path.join(backup_path, 'zcat.fits')
        
        if os.path.isfile(mtl_file) and os.path.isfile(zcat_file):
            return True
        else:
            return False

        
    def backup_epoch_data(self, epoch_id=0):
        """
        Deletes files in the temporary output directory

        Args:
            epoch_id (int): Epoch's ID to backup/copy from the output directory.

        """
        backup_path = os.path.join(self.output_path, '{}'.format(epoch_id))

        # keep a copy of zcat.fits
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)

        # keep a copy of mtl.fits
        shutil.copy(self.mtl_file, backup_path)
        shutil.copy(self.zcat_file, backup_path)

        # keep a copy of all the fiberassign files
        fiber_backup_path = os.path.join(backup_path, 'fiberassign')
        if not os.path.exists(fiber_backup_path):
            os.makedirs(fiber_backup_path)

        for tilefile in self.tilefiles:
            shutil.copy(tilefile, fiber_backup_path)


    def create_surveyfile(self, epoch):
        """
        Creates text file of tiles survey_list.txt to be used by fiberassign

        Args:
            epoch (int) : epoch of tiles to write

        Notes:
            The file is written to the temporary directory in self.tmp_output_path
        """

        # create survey list from mtl_epochs IDS
        surveyfile = os.path.join(self.tmp_output_path, "survey_list.txt")
        tiles      = self.epoch_tiles[epoch]

        np.savetxt(surveyfile, tiles, fmt='%d')

        print("{} tiles to be included in fiberassign".format(len(tiles)))

        return  len(tiles)
        
    def update_observed_tiles(self, epoch):
        """
        Creates the list of tilefiles to be gathered to build the redshift catalog.
        """        
        self.tilefiles = list()

        tiles = self.epoch_tiles[epoch]

        for i in tiles:
            tilename    = os.path.join(self.tmp_fiber_path, 'fiberassign-{:06d}.fits'.format(i))

            # retain ability to use previous version of tile files
            oldtilename = os.path.join(self.tmp_fiber_path, 'tile-%05d.fits'.format(i))

            if os.path.isfile(tilename):
                self.tilefiles.append(tilename)

            elif os.path.isfile(oldtilename):
                self.tilefiles.append(oldtilename)

            else:
              print('Suggested but does not exist {}'.format(tilename))

        print("{} {} tiles to gather for zcat from {}.".format(asctime(), len(self.tilefiles), self.tmp_fiber_path))


    def simulate_epoch(self, epoch, truth, targets, exposures, perfect=False, zcat=None):
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

        # create the MTL file
        print("{} Starting MTL".format(asctime()))

        self.mtl_file = os.path.join(self.tmp_output_path, 'mtl.fits')

        if (zcat is None):
          mtl = desitarget.mtl.make_mtl(targets, obscon='DARK|GRAY')
            
        elif len(zcat) == 0:
          mtl = desitarget.mtl.make_mtl(targets, obscon='DARK|GRAY')

        else:
          mtl = desitarget.mtl.make_mtl(targets, obscon='DARK|GRAY', zcat=zcat)

        print("{} Writing MTL".format(asctime()))

        
        # fits = FITS(self.mtl_file, 'rw') 
        # fits.write(mtl.as_array())
        # fits.close()

        # log.warning('Not writing over .mtl.')

        if not os.path.exists(self.mtl_file):
          mtl.write(self.mtl_file, overwrite=False)

        del mtl

        gc.collect()

        print("{} Finished MTL".format(asctime()))
                
        # clean files and prepare fiberasign inputs
        tilefiles = sorted(glob.glob(self.tmp_fiber_path+'/tile*.fits'))

        if tilefiles:
            for tilefile in tilefiles:
                os.remove(tilefile)

        # setup the tileids for the current observation epoch
        ntodo = self.create_surveyfile(epoch)

        if ntodo > 0:        
          # launch fiberassign
          print("{} Launching fiberassign".format(asctime()))

          ##  f = open('fiberassign.log', 'a')

          # '--stdstar',  self.stdfile  '--fibstatusfile',  self.fibstatusfile],
          # '--overwrite'

          # log.warning('Not overwriting fiberassign output')
          overwrite = False

          cmd = [self.fiberassign,
                 '--mtl',  os.path.join(self.tmp_output_path, 'mtl.fits'),
                 '--sky',  self.skyfile,
                 '--surveytiles',  self.surveyfile,
                 '--rundate', self.assigndates[epoch],
                 '--outdir', os.path.join(self.tmp_output_path, 'fiberassign')]

          if overwrite:
              cmd = cmd + ['--overwrite']

          if self.footprintfile is not None:
              cmd = cmd + ['--footprint', self.footprintfile]
          
          cmd = ' '.join(x for x in cmd)
          
          print(cmd)
          
          p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE) 

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
        
          print("{} Finished fiberassign".format(asctime()))

          ##  f.close()

        else:
          print("Skipping fiberassign for {} tiles.".format(ntodo))
          
        # create a list of fiberassign tiles to read and update zcat.
        self.update_observed_tiles(epoch)

        print('tilefiles', len(self.tilefiles))
        
        # write the zcat, it uses the tilesfiles constructed in the last step.
        self.zcat_file = os.path.join(self.tmp_output_path, 'zcat.fits')

        print("{} starting quickcat".format(asctime()))

        newzcat, fibermaps = quickcat(self.tilefiles, targets, truth, zcat=zcat,
                                      exposures=exposures, perfect=perfect)

        if fibermaps is not None:
          for key in fibermaps.keys():            
            (fpath, fmap) = fibermaps[key]

            print('Writing {}.'.format(fpath))
            
            fpath         = self.output_path + '/{}/fiberassign/{}'.format(epoch, fpath)

            dirname       = os.path.dirname(fpath) 
 
            if not os.path.exists(fpath):
              Path(dirname).mkdir(parents=True, exist_ok=True)
         
              fmap.write(fpath, format='fits', overwrite=True)

        if not os.path.exists(self.zcat_file):
          print("{} writing zcat".format(asctime()))

          newzcat.write(self.zcat_file, format='fits', overwrite=True)

        print("{} Finished zcat".format(asctime()))

        del newzcat

        gc.collect()

        return 


    def simulate(self):
        """
        Simulate the DESI setup described by a SimSetup object.
        """
        self.create_directories()

        exposures = self.exposures
        
        print('Reading truth.')
        
        truth = fitsio.read(os.path.join(self.targets_path,'truth.fits'))
        
        print('Reading targets.')

        targets = fitsio.read(os.path.join(self.targets_path,'targets.fits'))

        # Cut to observed tiles.        
        isin    = is_point_in_desi(self.tiles, targets['RA'], targets['DEC']) 

        targets = targets[isin]
        truth   =   truth[isin] 

        # print(isin)
        # print(len(targets), len(truth))
        
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
            print('--- Epoch {} ---'.format(epoch))
            
            if not self.epoch_data_exists(epoch_id=epoch):       
                # Initializes mtl and zcat
                if epoch == 0:
                    zcat = None

                else:
                    print('INFO: Running Epoch {}'.format(epoch))
                    print('INFO: reading zcat from previous epoch')
                    epochdir = os.path.join(self.output_path, str(epoch - 1))
                    zcat     = Table.read(os.path.join(epochdir, 'zcat.fits'))

                # Update mtl and zcat.
                self.simulate_epoch(epoch, truth, targets, exposures, perfect=True, zcat=zcat)

                # copy mtl and zcat to epoch directory
                self.backup_epoch_data(epoch_id=epoch)

                del zcat

                gc.collect()
                
            else:
                print('--- Epoch {} Already Exists ---'.format(epoch))
                
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
